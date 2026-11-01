{{- define "relsynth.name" -}}relsynth{{- end -}}
{{- define "relsynth.fullname" -}}{{- printf "%s-relsynth" .Release.Name | trunc 50 | trimSuffix "-" -}}{{- end -}}
{{- define "relsynth.image" -}}{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}{{- end -}}
{{- define "relsynth.labels" -}}
app.kubernetes.io/name: relsynth
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
{{- define "relsynth.selector" -}}
app.kubernetes.io/name: relsynth
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
{{- define "relsynth.sa" -}}{{ include "relsynth.fullname" . }}{{- end -}}
{{/* the migrate job runs as a pre-install hook, before the release's own
     ServiceAccount exists, so it gets a hook ServiceAccount of its own */}}
{{- define "relsynth.migrateSa" -}}
{{- if .Values.serviceAccount.create -}}{{ include "relsynth.fullname" . }}-migrate{{- else -}}{{ include "relsynth.sa" . }}{{- end -}}
{{- end -}}

{{/* environment shared by every pod */}}
{{- define "relsynth.env" -}}
- name: RELSYNTH_SVC_DB_URL
  valueFrom: {secretKeyRef: {name: {{ .Values.database.existingSecret }}, key: {{ .Values.database.key }}}}
- name: RELSYNTH_SVC_DATA_DIR
  value: /var/lib/relsynth/data
- name: RELSYNTH_SVC_MASTER_KEY_REF
  value: file:/run/secrets/relsynth-master/{{ .Values.masterKey.key }}
- name: RELSYNTH_SVC_SECRET_PREFIX_FILE
  value: file:/run/secrets/sources/
- name: RELSYNTH_SVC_LOG_JSON
  value: "true"
{{- range $k, $v := .Values.config }}
{{- if ne (toString $v) "" }}
- name: RELSYNTH_SVC_{{ upper $k }}
  value: {{ toString $v | quote }}
{{- end }}
{{- end }}
{{- with .Values.extraEnv }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{- define "relsynth.podSecurity" -}}
{{- $ctx := deepCopy .Values.podSecurityContext -}}
{{- if .Values.pinUid }}{{- $_ := set $ctx "runAsUser" 10001 }}{{- $_ := set $ctx "runAsGroup" 0 }}{{- $_ := set $ctx "fsGroup" 0 }}{{- end -}}
{{ toYaml $ctx }}
{{- end -}}

{{/* volumes: master key, shared data, per-tenant data */}}
{{- define "relsynth.volumes" -}}
- name: master-key
  secret: {secretName: {{ .Values.masterKey.existingSecret }}, defaultMode: 0440}
- name: data
  persistentVolumeClaim: {claimName: {{ .Values.storage.existingClaim | default (printf "%s-data" (include "relsynth.fullname" .)) }}}
- name: tmp
  emptyDir: {}
{{- end -}}
{{- define "relsynth.mounts" -}}
- {name: master-key, mountPath: /run/secrets/relsynth-master, readOnly: true}
- {name: data, mountPath: /var/lib/relsynth}
- {name: tmp, mountPath: /tmp}
{{- end -}}
